// Dumps Guitar Pro files (.gp3/4/5, .gpx, .gp) to JSON lines with TuxGuitar's
// own readers: one line per file with its tracks, tempo map and every note's
// string, fret, timing and playing technique. See eval/tabdump.sh.
import app.tuxguitar.io.base.TGSongReader;
import app.tuxguitar.io.base.TGSongReaderHandle;
import app.tuxguitar.song.factory.TGFactory;
import app.tuxguitar.song.models.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;

public class TabDump {
    static String q(String s) {
        if (s == null) return "\"\"";
        StringBuilder b = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            if (c == '"' || c == '\\') b.append('\\').append(c);
            else if (c < 32) b.append(String.format("\\u%04x", (int) c));
            else b.append(c);
        }
        return b.append('"').toString();
    }

    static TGSongReader readerFor(String path) throws Exception {
        String p = path.toLowerCase();
        if (p.endsWith(".gp")) return new app.tuxguitar.io.gpx.v7.GPXInputStream();
        if (p.endsWith(".gpx")) return new app.tuxguitar.io.gpx.v6.GPXInputStream();
        app.tuxguitar.io.gtp.GTPSettings st = new app.tuxguitar.io.gtp.GTPSettings();
        if (p.endsWith(".gp5")) return new app.tuxguitar.io.gtp.GP5InputStream(st);
        if (p.endsWith(".gp4")) return new app.tuxguitar.io.gtp.GP4InputStream(st);
        if (p.endsWith(".gp3")) return new app.tuxguitar.io.gtp.GP3InputStream(st);
        return null;
    }

    public static void main(String[] args) throws Exception {
        PrintStream out = new PrintStream(new FileOutputStream(args[0]), false, "UTF-8");
        for (int i = 1; i < args.length; i++) {
            String path = args[i];
            try {
                TGSongReader reader = readerFor(path);
                if (reader == null) continue;
                TGSongReaderHandle h = new TGSongReaderHandle();
                h.setFactory(new TGFactory());
                h.setInputStream(new BufferedInputStream(new FileInputStream(path)));
                reader.read(h);
                TGSong song = h.getSong();
                StringBuilder sb = new StringBuilder();
                sb.append("{\"path\":").append(q(path));
                sb.append(",\"title\":").append(q(song.getName()));
                sb.append(",\"artist\":").append(q(song.getArtist()));
                sb.append(",\"album\":").append(q(song.getAlbum()));
                // Bars: start tick, length, tempo (quarter notes per minute), time signature.
                sb.append(",\"bars\":[");
                Iterator<TGMeasureHeader> hs = song.getMeasureHeaders();
                boolean first = true;
                while (hs.hasNext()) {
                    TGMeasureHeader mh = hs.next();
                    if (!first) sb.append(',');
                    first = false;
                    sb.append('[').append(mh.getStart()).append(',').append(mh.getLength()).append(',')
                      .append(mh.getTempo().getQuarterValue()).append(',')
                      .append(mh.getTimeSignature().getNumerator()).append(',')
                      .append(mh.getTimeSignature().getDenominator().getValue()).append(',')
                      .append(mh.getRepeatClose()).append(',').append(mh.isRepeatOpen() ? 1 : 0).append(']');
                }
                sb.append("],\"tracks\":[");
                Iterator<TGTrack> ts = song.getTracks();
                boolean ft = true;
                while (ts.hasNext()) {
                    TGTrack t = ts.next();
                    if (!ft) sb.append(',');
                    ft = false;
                    TGChannel ch = null;
                    Iterator<TGChannel> chs = song.getChannels();
                    while (chs.hasNext()) {
                        TGChannel c = chs.next();
                        if (c.getChannelId() == t.getChannelId()) ch = c;
                    }
                    sb.append("{\"name\":").append(q(t.getName()));
                    sb.append(",\"program\":").append(ch == null ? -1 : ch.getProgram());
                    sb.append(",\"percussion\":").append(t.isPercussion());
                    sb.append(",\"offset\":").append(t.getOffset());
                    sb.append(",\"strings\":[");
                    List<TGString> strs = new ArrayList<>(t.getStrings());
                    strs.sort(Comparator.comparingInt(TGString::getNumber));
                    for (int s = 0; s < strs.size(); s++) {
                        if (s > 0) sb.append(',');
                        sb.append(strs.get(s).getValue());
                    }
                    // notes: [tick, duration ticks, voice, string (1 = highest), fret, flags]
                    sb.append("],\"notes\":[");
                    boolean fn = true;
                    Iterator<TGMeasure> ms = t.getMeasures();
                    while (ms.hasNext()) {
                        TGMeasure m = ms.next();
                        for (int b = 0; b < m.countBeats(); b++) {
                            TGBeat beat = m.getBeat(b);
                            for (int v = 0; v < beat.countVoices(); v++) {
                                TGVoice voice = beat.getVoice(v);
                                if (voice == null || voice.isEmpty() || voice.isRestVoice()) continue;
                                long dur = voice.getDuration().getTime();
                                for (TGNote n : voice.getNotes()) {
                                    TGNoteEffect e = n.getEffect();
                                    String flags = (n.isTiedNote() ? "t" : "") + (e.isDeadNote() ? "x" : "")
                                        + (e.isPalmMute() ? "m" : "") + (e.isHarmonic() ? "h" : "") + (e.isBend() ? "b" : "")
                                        + (e.isSlide() ? "s" : "") + (e.isHammer() ? "o" : "") + (e.isVibrato() ? "v" : "")
                                        + (e.isLetRing() ? "r" : "") + (e.isGhostNote() ? "g" : "") + (e.isGrace() ? "c" : "")
                                        + (e.isTapping() ? "p" : "") + (e.isTremoloPicking() ? "k" : "");
                                    if (!fn) sb.append(',');
                                    fn = false;
                                    sb.append('[').append(beat.getStart()).append(',').append(dur).append(',').append(v).append(',')
                                      .append(n.getString()).append(',').append(n.getValue()).append(',').append(q(flags)).append(']');
                                }
                            }
                        }
                    }
                    sb.append("]}");
                }
                sb.append("]}");
                out.println(sb);
            } catch (Throwable ex) {
                System.err.println("FAILED " + path + ": " + ex);
            }
        }
        out.close();
    }
}
